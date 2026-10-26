"""Apply specificity-guided fields to the complete real peak atlas.

Scores use linear CPM in the celltype-by-stage matrix. Neighborhoods and both
embeddings come from the published Leiden-pseudobulk atlas. Paths are an explicit
orientation of fixed peaks; a specificity rate does not identify physical motion.
No biological uncertainty is inferred from these pooled pseudobulks.
"""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
import warnings
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts/chrom_velo'))
from real_specialization import parse_condition_grid, to_condition_cube, compute_block_statistics
from specialization_flow import bidirectional_rate_guidance, aggregate_specificity
from real_sparse_projection import block_structural_flow as structural_flow

DATA = ROOT.parent / 'data/annotated_data/objects_v2'
DEFAULT_STATE = DATA / 'peaks_by_ct_tp_master.h5ad'
DEFAULT_ATLAS = DATA / 'peaks_by_pb_annotated_master.h5ad'
DEFAULT_OUT = HERE / 'outputs/real_specialization_v1'
PARAMS = dict(min_cells=20, min_total_counts=20, edge_tolerance=.01,
              gradient_power=1., min_neighbors=2, min_coherence=.25,
              min_parameter_cosine=.8, rate_effect_floor=.002)


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def read_column(group, name):
    g = group[name]
    if isinstance(g, h5py.Dataset):
        return g.asstr()[:] if g.dtype.kind in 'OSU' else g[:]
    cat = g['categories'].asstr()[:] if g['categories'].dtype.kind in 'OSU' else g['categories'][:]
    codes = g['codes'][:]
    if np.any(codes < 0):
        values = np.full(len(codes), '', dtype=object)
        values[codes >= 0] = cat[codes[codes >= 0]]
        return values
    return cat[codes]


def read_frame(group, columns=None):
    index = pd.Index(read_column(group, group.attrs['_index']).astype(str))
    if columns is None:
        columns = [k for k in group if k != group.attrs['_index']]
    return pd.DataFrame({k: read_column(group, k) for k in columns if k in group}, index=index)


def read_graph(group):
    return sparse.csr_matrix((group['data'][:], group['indices'][:], group['indptr'][:]),
                             shape=tuple(group.attrs['shape']))


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, data):
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + '\n')


def cosine(a, b):
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a*b, axis=1), den, out=np.full(len(a), np.nan), where=den > 1e-12)


def corr(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[good], b[good]).statistic) if good.sum() > 2 else np.nan


def score_rate(result):
    return result['score'], result.get('rate', result.get('interval_rate'))


def load_and_score(state_path, atlas_path, output, block_size):
    with h5py.File(state_path) as state, h5py.File(atlas_path) as atlas:
        var = read_frame(state['var'])
        grid = parse_condition_grid(var)
        state_ids = pd.Index(read_column(state['obs'], state['obs'].attrs['_index']).astype(str))
        obs = read_frame(atlas['obs'], ['celltype', 'lineage', 'timepoint', 'leiden_coarse',
            'nearest_gene', 'linked_gene', 'associated_gene', 'chrom', 'start', 'end',
            'peak_type', 'peak_type_argelaguet', 'distance_to_tss'])
        if not state_ids.is_unique or not obs.index.is_unique or not state_ids.equals(obs.index):
            raise ValueError('Real state and atlas must have identical unique ordered peak IDs')
        coords = {'umap2d': atlas['obsm/X_umap_2D'][:], 'umap3d': atlas['obsm/X_umap_3D'][:]}
        for key, source in [('umap2d','X_umap_2D'), ('umap3d','X_umap_3D')]:
            if not np.array_equal(coords[key], state['obsm'][source][:]):
                raise ValueError(f'Copied state embedding differs from atlas {source}')
        graph = read_graph(atlas['obsp/connectivities'])
        n, p = state['layers/normalized'].shape
        common_scale = float(state['uns/common_scale_factor'][()])
        cpm_scale = 1e6 / common_scale
        celltypes = list(grid['celltypes']); times = np.asarray(grid['stages'], dtype=float)
        observed = np.asarray(grid['present']) & (np.asarray(grid['n_cells']) >= PARAMS['min_cells'])
        common = observed.all(axis=1)
        if common.sum() < 2: raise ValueError('Insufficient all-stage common celltype support')
        cpm = np.empty((n,p), dtype=np.float32)
        opening = np.full((n,p), np.nan, dtype=np.float32)
        arrays = {}
        for method in ['tau','entropy']:
            for key, width in [('specificity',6),('rate',5),('available_specificity',6),
                ('available_rate',5),('shrinkage05_rate',5),('min_cells50_specificity',6),('min_cells50_rate',5)]:
                arrays[f'{method}_{key}'] = np.full((n,width), np.nan, dtype=np.float32)
        arrays['fixed_raw_total'] = np.zeros((n,6), dtype=np.float32)
        arrays['fixed_mean_cpm'] = np.zeros((n,6), dtype=np.float32)
        min_detection = np.zeros(n, dtype=np.float32)
        max_norm_error = 0.
        for start in range(0,n,block_size):
            stop = min(start+block_size,n); sl = slice(start,stop)
            norm = state['layers/normalized'][sl]
            raw = state['layers/sum'][sl]
            if not np.isfinite(norm).all() or np.any(norm < 0): raise ValueError('Invalid normalized counts')
            expected = raw * var['scale_factor'].to_numpy()[None,:]
            max_norm_error = max(max_norm_error, float(np.max(np.abs(expected - norm))))
            x = (norm*cpm_scale).astype(np.float32); cpm[sl] = x
            stats = compute_block_statistics(x, raw, grid, min_cells=20, min_total_counts=20)
            shrink = compute_block_statistics(x, raw, grid, min_cells=20, min_total_counts=20, shrinkage=.05)
            cells50 = compute_block_statistics(x, raw, grid, min_cells=50, min_total_counts=20)
            for method in ['tau','entropy']:
                for panel, prefix in [('fixed_common',''),('available','available_')]:
                    score, rate = score_rate(stats[panel][method])
                    arrays[f'{method}_{prefix}specificity'][sl] = score
                    arrays[f'{method}_{prefix}rate'][sl] = rate
                arrays[f'{method}_shrinkage05_rate'][sl] = score_rate(shrink['fixed_common'][method])[1]
                score, rate = score_rate(cells50['fixed_common'][method])
                arrays[f'{method}_min_cells50_specificity'][sl] = score
                arrays[f'{method}_min_cells50_rate'][sl] = rate
            for c in range(len(celltypes)):
                for t in range(5):
                    col = grid['column_index'][c,t]
                    if col >= 0: opening[sl,col] = stats['opening_rate'][:,c,t]
            rc = to_condition_cube(raw, grid)[:,common]
            xc = to_condition_cube(x, grid)[:,common]
            arrays['fixed_raw_total'][sl] = np.nansum(rc, axis=1)
            arrays['fixed_mean_cpm'][sl] = np.nanmean(xc,axis=1)
            min_detection[sl] = np.mean(xc > 0,axis=1).min(axis=1)
            if start == 0 or stop == n or start//block_size % 10 == 0:
                log(f'Scored {stop:,}/{n:,} peaks; fixed panel {common.sum()} celltypes')
        if max_norm_error > 1e-6:
            raise ValueError(f'Normalization identity failed: max error {max_norm_error}')
        a = ad.AnnData(cpm,obs=obs,var=var)
        a.layers['accessibility_rate'] = opening
        a.obsm.update({f'X_{k}':v for k,v in coords.items()})
        a.obsm.update(arrays)
        a.obsp['atlas_connectivities'] = graph.astype(np.float32)
        a.obs['min_fixed_detection_fraction'] = min_detection
        valid_all = np.isfinite(arrays['tau_specificity']).all(axis=1)
        max_tau = np.max(np.where(np.isfinite(arrays['tau_specificity']), arrays['tau_specificity'], np.inf), axis=1)
        min_cpm = arrays['fixed_mean_cpm'].min(axis=1)
        threshold_tau = float(np.quantile(max_tau[valid_all],.05))
        threshold_cpm = float(np.median(min_cpm[valid_all]))
        a.obs['broad_reference'] = valid_all & (max_tau <= threshold_tau) & (min_cpm >= threshold_cpm) & (min_detection >= .8)
        a.uns['real_specialization'] = dict(parameters=PARAMS, state_path=str(state_path), atlas_path=str(atlas_path),
            state_units='CPM', cpm_scale=cpm_scale, common_scale_factor=common_scale,
            celltypes=celltypes, stages=times, common_celltypes=np.asarray(celltypes)[common].tolist(),
            condition_n_cells=np.asarray(grid['n_cells']), condition_valid=observed,
            normalization_max_abs_error=max_norm_error, specificity_weighting='terminal',
            primary_panel='celltypes with at least 20 cells at every one of the six stages',
            uncertainty='No replicate confidence intervals; parameter sensitivity only',
            broad_reference_definition='bottom 5% of maximum stage tau, above median minimum-stage mean CPM, detection >=80% at every stage; fixed panel only',
            broad_reference_max_tau=threshold_tau,broad_reference_min_cpm=threshold_cpm,
            graph_origin='published Leiden-pseudobulk atlas connectivities; UMAP coordinates retained exactly',
            opening_rate_definition='CPM/hpf within celltype; both groups >=20 cells; combined raw count >=20; missing and terminal entries NaN')
        return a, grid


def fit_fields(a, output):
    p = PARAMS
    graph = a.obsp['atlas_connectivities']
    embeddings = {d:a.obsm[f'X_{d}'] for d in ['umap2d','umap3d']}
    sensitivity = []
    for method in ['tau','entropy']:
        score = a.obsm[f'{method}_specificity'][:,-1]
        rates = a.obsm[f'{method}_rate']
        log(f'Computing {method} paths on all {a.n_obs:,} peaks')
        fields = {}
        for direction, sign in [('up',1),('down',-1)]:
            fields[direction] = structural_flow(graph, sign*score, embeddings,
                edge_tolerance=p['edge_tolerance'],gradient_power=p['gradient_power'],min_support=p['min_neighbors'])
            log(f'{method} {direction}: primary field projected')
            f = fields[direction]
            a.obsp[f'{method}_{direction}_transitions'] = f['probabilities'].astype(np.float32)
            a.obs[f'{method}_{direction}_support'] = f['support_count'].astype(np.int32)
            a.obs[f'{method}_{direction}_support_fraction'] = f['support_fraction'].astype(np.float32)
            for d in embeddings:
                a.obsm[f'{method}_{direction}_{d}'] = f['vectors'][d].astype(np.float32)
                a.obs[f'{method}_{direction}_{d}_coherence'] = f['coherence'][d].astype(np.float32)
        cosines = {(direction,d):[] for direction in fields for d in embeddings}
        for tol, power in [(.005,1.),(.02,1.),(.01,2.)]:
            for direction, sign in [('up',1),('down',-1)]:
                alt = structural_flow(graph,sign*score,embeddings,edge_tolerance=tol,gradient_power=power,min_support=p['min_neighbors'])
                log(f'{method} {direction}: tolerance={tol}, power={power} projected')
                for d in embeddings:
                    cs = cosine(fields[direction]['vectors'][d],alt['vectors'][d])
                    cosines[(direction,d)].append(cs)
                    supported = (fields[direction]['coherence'][d]>=p['min_coherence']) & np.isfinite(cs)
                    sensitivity.append(dict(method=method,direction=direction,embedding=d,edge_tolerance=tol,
                        gradient_power=power,reference_supported=int(supported.sum()),
                        median_cosine=float(np.nanmedian(cs[supported])) if supported.any() else np.nan,
                        fraction_cosine_above_08=float(np.mean(cs[supported]>=.8)) if supported.any() else np.nan,
                        alternative_active=int(alt['active'].sum())))
                del alt; gc.collect()
        for d in embeddings:
            masks = {}
            for direction in fields:
                cs = np.asarray(cosines[(direction,d)])
                # Missing alternative direction counts as zero agreement; never
                # hide a parameter setting that eliminates the reference arrow.
                stability = np.mean(np.where(np.isfinite(cs),cs,0.),axis=0)
                mask = (fields[direction]['support_count']>=p['min_neighbors']) & (fields[direction]['coherence'][d]>=p['min_coherence']) & (stability>=p['min_parameter_cosine'])
                masks[direction] = mask
                a.obs[f'{method}_{direction}_{d}_stability'] = stability.astype(np.float32)
                a.obs[f'{method}_{direction}_{d}_display'] = mask
            guided = np.stack([bidirectional_rate_guidance(
                fields['up']['vectors'][d],fields['down']['vectors'][d],rates[:,t],
                valid=(np.isfinite(rates[:,t]) &
                    (a.obsm[f'{method}_thinning_sign_agreement'][:,t] >= a.uns['real_specialization']['depth_matching']['min_thinning_sign_agreement'])
                    if 'depth_matching' in a.uns['real_specialization'] and 'min_thinning_sign_agreement' in a.uns['real_specialization']['depth_matching']
                    else np.isfinite(rates[:,t])),
                upward_valid=masks['up'],downward_valid=masks['down'],
                min_coherence=p['min_coherence'],rate_floor=p['rate_effect_floor']) for t in range(5)],axis=1)
            a.obsm[f'{method}_guided_{d}'] = guided.reshape(a.n_obs,-1).astype(np.float32)
        del fields; gc.collect()
    pd.DataFrame(sensitivity).to_csv(output/'parameter_sensitivity.csv',index=False)


def summarize(a, output):
    summary = dict(n_peaks=a.n_obs,n_conditions=a.n_vars,common_celltypes=list(a.uns['real_specialization']['common_celltypes']),
        n_broad_reference=int(a.obs['broad_reference'].sum()),parameters=PARAMS,methods={})
    rows=[]; cluster_rows=[]
    for method in ['tau','entropy']:
        scores=a.obsm[f'{method}_specificity']; rates=a.obsm[f'{method}_rate']
        alternative=a.obsm[f'{method}_available_specificity']
        rawtotal=a.obsm['fixed_raw_total']
        sm=dict(valid_score_by_stage=np.isfinite(scores).sum(0),valid_rate_by_interval=np.isfinite(rates).sum(0),
            terminal_score_quantiles=np.nanquantile(scores[:,-1],[0,.1,.25,.5,.75,.9,1]),
            available_terminal_rank_correlation=corr(scores[:,-1],alternative[:,-1]),
            min_cells50_terminal_rank_correlation=corr(scores[:,-1],a.obsm[f'{method}_min_cells50_specificity'][:,-1]),
            terminal_vs_duration_rank_correlation=corr(scores[:,-1],aggregate_specificity(scores,a.uns['real_specialization']['stages'],weighting='duration')['aggregate']))
        for d in ['umap2d','umap3d']:
            dim=int(d[-2]);guided=a.obsm[f'{method}_guided_{d}'].reshape(a.n_obs,5,dim)
            sm[f'displayed_structural_{d}']=int(a.obs[f'{method}_up_{d}_display'].sum())
            sm[f'displayed_temporal_{d}']=np.sum(np.linalg.norm(guided,axis=2)>0,axis=0)
        for t in range(5):
            r=rates[:,t]; eligible=np.isfinite(r)&(np.abs(r)>=PARAMS['rate_effect_floor'])
            sr=a.obsm[f'{method}_shrinkage05_rate'][:,t]
            er=a.obsm[f'{method}_min_cells50_rate'][:,t]
            for label,other in [('shrinkage_005',sr),('min_cells50_changed_panel',er),('available_pair_changed_panel',a.obsm[f'{method}_available_rate'][:,t])]:
                both=eligible&np.isfinite(other)
                rows.append(dict(method=method,interval=t,comparison=label,n_compared=int(both.sum()),
                    sign_agreement=float(np.mean(np.sign(r[both])==np.sign(other[both]))) if both.any() else np.nan,
                    spearman=corr(r,other)))
            sm.setdefault('rate_positive_by_interval',[]).append(int((eligible&(r>0)).sum()))
            sm.setdefault('rate_negative_by_interval',[]).append(int((eligible&(r<0)).sum()))
            sm.setdefault('supported_at_count50_by_interval',[]).append(int((eligible&(rawtotal[:,t]>=50)&(rawtotal[:,t+1]>=50)).sum()))
        summary['methods'][method]=sm
        for cluster, ids in a.obs.groupby('leiden_coarse',observed=True).indices.items():
            for t in range(5):
                r=rates[ids,t];v=a.obsm[f'{method}_guided_umap2d'][ids].reshape(-1,5,2)[:,t]
                valid=np.isfinite(r); active=np.linalg.norm(v,axis=1)>0
                cluster_rows.append(dict(method=method,leiden_coarse=str(cluster),interval=t,
                    start_hpf=float(a.uns['real_specialization']['stages'][t]),end_hpf=float(a.uns['real_specialization']['stages'][t+1]),
                    n_peaks=len(ids),n_supported=int(valid.sum()),n_guided=int(active.sum()),
                    mean_specificity_rate=float(np.mean(r[valid])) if valid.any() else np.nan,
                    mean_vector_x=float(np.mean(v[active,0])) if active.any() else np.nan,
                    mean_vector_y=float(np.mean(v[active,1])) if active.any() else np.nan))
    pd.DataFrame(rows).to_csv(output/'score_sensitivity.csv',index=False)
    pd.DataFrame(cluster_rows).to_csv(output/'cluster_summary.csv',index=False)
    # Direct accessibility changes in biologically relevant annotated loci are
    # inspection targets, not ground truth or guaranteed signs.
    genes=['slc4a1a','cdx1a','phox2bb','gata1a','tal1','sox10']
    locus_rows=[]
    for gene in genes:
        match=np.zeros(a.n_obs,dtype=bool)
        for column in ['nearest_gene','linked_gene','associated_gene']:
            if column in a.obs:
                match|=a.obs[column].astype(str).str.split('[,;| ]').apply(lambda x:gene in x).to_numpy()
        ids=np.flatnonzero(match)
        for col in range(a.n_vars):
            r=a.layers['accessibility_rate'][ids,col];valid=np.isfinite(r)
            if valid.any():
                locus_rows.append(dict(gene=gene,condition=str(a.var_names[col]),celltype=str(a.var.iloc[col]['annotation_ML_coarse']),stage=str(a.var.iloc[col]['dev_stage']),
                    n_annotated_peaks=len(ids),n_supported=int(valid.sum()),mean_cpm_per_hpf=float(r[valid].mean()),median_cpm_per_hpf=float(np.median(r[valid]))))
    pd.DataFrame(locus_rows).to_csv(output/'gene_locus_opening_summary.csv',index=False)
    write_json(output/'run_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,default=DEFAULT_STATE)
    parser.add_argument('--atlas',type=Path,default=DEFAULT_ATLAS)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--block-size',type=int,default=512)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/'real_specialization_flow.h5ad'
    if args.render_only:
        a=ad.read_h5ad(target);grid=parse_condition_grid(a.var)
    else:
        a,grid=load_and_score(args.state,args.atlas,args.output_dir,args.block_size)
        fit_fields(a,args.output_dir)
        summary=summarize(a,args.output_dir)
        log('Saving full-atlas results')
        a.write_h5ad(target,compression='gzip')
        log('Saved '+str(target))
    from real_specialization_report import render_report
    render_report(a,grid,args.output_dir)
    log('Real-data analysis and visualizations complete')


if __name__=='__main__':
    main()
