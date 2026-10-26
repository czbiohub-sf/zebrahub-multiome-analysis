"""Depth-matched specificity fields on the existing full peak atlas.

The original run remains a comparison. Matching counts across stages within each
celltype addresses an observed sampling-depth confound, not biological variance.
Measured CPM and within-celltype accessibility rates are retained unchanged.
"""
from pathlib import Path
import argparse
import shutil
import sys
import time
import h5py
import numpy as np
import pandas as pd
import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts/chrom_velo'))
from real_specialization import parse_condition_grid
from specialization_flow import bidirectional_rate_guidance
from depth_matched_specialization import estimate_depth_matched_block
from real_specialization_benchmark import (PARAMS, fit_fields, write_json, log,
                                          corr, DEFAULT_STATE)

DEFAULT_BASELINE = HERE / 'outputs/real_specialization_v1'
DEFAULT_OUT = HERE / 'outputs/real_specialization_depth_matched_v1'


def replace_scores(a, state_path, block_size=512, seed=4242, n_thinnings=5):
    grid = parse_condition_grid(a.var)
    for method in ['tau', 'entropy']:
        for key in ['specificity', 'rate']:
            a.obsm[f'{method}_uncorrected_{key}'] = a.obsm[f'{method}_{key}'].copy()
            a.obsm[f'{method}_{key}'][:] = np.nan
        a.obsm[f'{method}_thinning_sign_agreement'] = np.full((a.n_obs, 5), np.nan, dtype='float32')
    for key in ['mean', 'min']:
        a.obsm[f'depth_matched_{key}_raw_total'] = np.empty((a.n_obs, 6), dtype='float32')
    with h5py.File(state_path) as source:
        source_index = source['obs'][source['obs'].attrs['_index']].asstr()[:]
        if not np.array_equal(source_index, a.obs_names.astype(str)):
            raise ValueError('Source and saved atlas peak order differ')
        for start in range(0, a.n_obs, block_size):
            stop = min(start + block_size, a.n_obs)
            result = estimate_depth_matched_block(source['layers/sum'][start:stop], grid,
                global_start=start, seed=seed, n_thinnings=n_thinnings,
                min_cells=PARAMS['min_cells'], min_total_counts=PARAMS['min_total_counts'])
            for method in ['tau', 'entropy']:
                a.obsm[f'{method}_specificity'][start:stop] = result[method]['score']
                a.obsm[f'{method}_rate'][start:stop] = result[method]['rate']
                a.obsm[f'{method}_thinning_sign_agreement'][start:stop] = result[method]['thinning_sign_agreement']
            for key in ['mean', 'min']:
                a.obsm[f'depth_matched_{key}_raw_total'][start:stop] = result[f'{key}_raw_total']
            if start == 0 or stop == a.n_obs or start // block_size % 100 == 0:
                log(f'Depth matched {stop:,}/{a.n_obs:,} peaks; {n_thinnings} thinnings')
    info = a.uns['real_specialization']
    info['specificity_estimator'] = 'depth_matched'
    info['depth_matching'] = dict(seed=seed, n_thinnings=n_thinnings, block_size=block_size,
        min_cells=PARAMS['min_cells'], min_total_counts=PARAMS['min_total_counts'],
        min_thinning_sign_agreement=.8,
        display_filter_note='Temporal arrows require sign agreement in at least 4/5 count-thinnings; conditional Monte Carlo stability, not biological confidence; scalar rates retained unchanged',
        global_start_convention='zero-based source row index; fixed blocks; final block may be shorter',
        target_coverage=result['target_coverage'], thinning_probabilities=result['thinning_probabilities'],
        celltypes=result['celltypes'],
        support='Every thinning must have at least 20 total counts across the fixed panel at each supported stage; both interval endpoints required',
        estimator=f'Mean tau and entropy across {n_thinnings} binomial count-thinnings, normalized by matched exposure; no pseudocount or shrinkage',
        uncertainty='Thinning sign agreement is conditional Monte Carlo stability, not a confidence interval or biological replication',
        scope='Matching is across stages within each celltype; target exposures still differ between celltypes')
    info['uncorrected_sensitivity_keys'] = [key for key in a.obsm if any(s in key for s in ['available_', 'min_cells50_', 'shrinkage05_'])]
    info['uncorrected_sensitivity_note'] = 'These inherited alternative-panel arrays belong to the original uncorrected estimate; they are not depth-matched sensitivity estimates'
    info['state_and_opening_note'] = 'X, fixed_raw_total, fixed_mean_cpm and accessibility_rate remain original measured CPM/count summaries; depth matching changes only specificity and its derived fields'
    info['uncertainty'] = 'No biological confidence intervals; graph parameter sensitivity and conditional thinning stability only'
    a.obs['uncorrected_broad_reference'] = a.obs['broad_reference'].to_numpy()
    scores = a.obsm['tau_specificity']
    valid = np.isfinite(scores).all(axis=1)
    if not valid.any():
        raise ValueError('No all-stage supported peaks after matching')
    max_tau = np.max(np.where(np.isfinite(scores), scores, np.inf), axis=1)
    min_cpm = a.obsm['fixed_mean_cpm'].min(axis=1)
    tau_threshold = float(np.quantile(max_tau[valid], .05))
    cpm_threshold = float(np.median(min_cpm[valid]))
    a.obs['broad_reference'] = valid & (max_tau <= tau_threshold) & (min_cpm >= cpm_threshold) & (a.obs['min_fixed_detection_fraction'].to_numpy() >= .8)
    info['broad_reference_max_tau'] = tau_threshold
    info['broad_reference_min_cpm'] = cpm_threshold
    info['broad_reference_definition'] = 'Bottom 5% of maximum stage depth-matched tau among all-stage supported peaks; above median minimum-stage original mean CPM; original detection >=80% at every stage; fixed panel only'
    return grid


def summarize_matched(a, out):
    info = a.uns['real_specialization']
    times = np.asarray(info['stages'])
    summary = dict(n_peaks=a.n_obs, n_conditions=a.n_vars,
        common_celltypes=list(info['common_celltypes']),
        n_broad_reference=int(a.obs['broad_reference'].sum()), parameters=PARAMS,
        specificity_estimator='depth_matched', depth_matching=info['depth_matching'], methods={})
    comparisons = []
    clusters = []
    for method in ['tau', 'entropy']:
        score = a.obsm[f'{method}_specificity']; rate = a.obsm[f'{method}_rate']
        oldscore = a.obsm[f'{method}_uncorrected_specificity']; oldrate = a.obsm[f'{method}_uncorrected_rate']
        agreement = a.obsm[f'{method}_thinning_sign_agreement']
        result = dict(valid_score_by_stage=np.isfinite(score).sum(0), valid_rate_by_interval=np.isfinite(rate).sum(0),
            uncorrected_valid_score_by_stage=np.isfinite(oldscore).sum(0),
            uncorrected_valid_rate_by_interval=np.isfinite(oldrate).sum(0),
            terminal_score_quantiles=np.nanquantile(score[:, -1], [0, .1, .25, .5, .75, .9, 1]),
            uncorrected_terminal_rank_correlation=corr(score[:, -1], oldscore[:, -1]))
        for d in ['umap2d', 'umap3d']:
            dim = int(d[-2]); guided = a.obsm[f'{method}_guided_{d}'].reshape(a.n_obs, 5, dim)
            result[f'displayed_structural_{d}'] = int(a.obs[f'{method}_up_{d}_display'].sum())
            result[f'displayed_temporal_{d}'] = np.sum(np.linalg.norm(guided, axis=2) > 0, axis=0)
        for t in range(5):
            r = rate[:, t]; old = oldrate[:, t]
            valid = np.isfinite(r); both = valid & np.isfinite(old)
            eligible = valid & (np.abs(r) >= PARAMS['rate_effect_floor'])
            row = dict(method=method, start_hpf=times[t], end_hpf=times[t+1],
                n_supported=int(valid.sum()), n_both_supported=int(both.sum()),
                corrected_positive_fraction=float(np.mean(r[valid] > 0)) if valid.any() else np.nan,
                corrected_median_rate=float(np.median(r[valid])) if valid.any() else np.nan,
                uncorrected_matched_support_positive_fraction=float(np.mean(old[both] > 0)) if both.any() else np.nan,
                uncorrected_matched_support_median_rate=float(np.median(old[both])) if both.any() else np.nan,
                sign_agreement_with_uncorrected=float(np.mean(np.sign(r[both]) == np.sign(old[both]))) if both.any() else np.nan,
                rate_rank_correlation=corr(r, old),
                n_above_effect_floor=int(eligible.sum()),
                fraction_effect_eligible_thinning_sign_agreement_ge08=float(np.mean(agreement[eligible, t] >= .8)) if eligible.any() else np.nan,
                median_thinning_sign_agreement=float(np.median(agreement[valid, t])) if valid.any() else np.nan)
            comparisons.append(row)
            result.setdefault('rate_positive_by_interval', []).append(int((eligible & (r > 0)).sum()))
            result.setdefault('rate_negative_by_interval', []).append(int((eligible & (r < 0)).sum()))
        summary['methods'][method] = result
        for cluster, ids in a.obs.groupby('leiden_coarse', observed=True).indices.items():
            for t in range(5):
                r = rate[ids, t]; valid = np.isfinite(r)
                v = a.obsm[f'{method}_guided_umap2d'][ids].reshape(-1, 5, 2)[:, t]
                active = np.linalg.norm(v, axis=1) > 0
                clusters.append(dict(method=method, leiden_coarse=str(cluster), interval=t,
                    start_hpf=times[t], end_hpf=times[t+1], n_peaks=len(ids),
                    n_supported=int(valid.sum()), n_guided=int(active.sum()),
                    mean_specificity_rate=float(np.mean(r[valid])) if valid.any() else np.nan,
                    mean_vector_x=float(np.mean(v[active, 0])) if active.any() else np.nan,
                    mean_vector_y=float(np.mean(v[active, 1])) if active.any() else np.nan))
    pd.DataFrame(comparisons).to_csv(out / 'depth_matching_comparison.csv', index=False)
    pd.DataFrame(clusters).to_csv(out / 'cluster_summary.csv', index=False)
    write_json(out / 'run_summary.json', summary)
    return summary



def refresh_guidance(a):
    """Recompute only temporal vectors from saved structural paths and scores."""
    dm = a.uns['real_specialization']['depth_matching']
    dm['min_thinning_sign_agreement'] = .8
    dm['display_filter_note'] = 'Temporal arrows require at least 80% thinning sign agreement; conditional Monte Carlo stability, not biological confidence; scalar rates retained unchanged'
    for method in ['tau', 'entropy']:
        rates = a.obsm[f'{method}_rate']
        agreement = a.obsm[f'{method}_thinning_sign_agreement']
        for d in ['umap2d', 'umap3d']:
            vectors = np.stack([bidirectional_rate_guidance(
                a.obsm[f'{method}_up_{d}'], a.obsm[f'{method}_down_{d}'], rates[:, t],
                valid=np.isfinite(rates[:, t]) & (agreement[:, t] >= dm['min_thinning_sign_agreement']),
                upward_valid=a.obs[f'{method}_up_{d}_display'].to_numpy(),
                downward_valid=a.obs[f'{method}_down_{d}_display'].to_numpy(),
                min_coherence=PARAMS['min_coherence'], rate_floor=PARAMS['rate_effect_floor'])
                for t in range(5)], axis=1)
            a.obsm[f'{method}_guided_{d}'] = vectors.reshape(a.n_obs, -1).astype('float32')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, default=DEFAULT_BASELINE)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--state', type=Path, default=DEFAULT_STATE)
    p.add_argument('--block-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=4242)
    p.add_argument('--n-thinnings', type=int, default=5)
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--refresh-guidance', action='store_true', help='Apply the thinning stability display gate to saved fields without rerunning score estimates')
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'real_specialization_flow.h5ad'
    if args.refresh_guidance:
        log('Refreshing temporal display with thinning sign-stability filter')
        a = ad.read_h5ad(target); grid = parse_condition_grid(a.var)
        refresh_guidance(a)
        summarize_matched(a, args.output_dir)
        a.write_h5ad(target, compression='gzip')
        log('Saved stability-filtered temporal fields')
    elif args.render_only:
        a = ad.read_h5ad(target); grid = parse_condition_grid(a.var)
    else:
        log('Loading completed baseline atlas')
        a = ad.read_h5ad(args.baseline_dir / 'real_specialization_flow.h5ad')
        grid = replace_scores(a, args.state, args.block_size, args.seed, args.n_thinnings)
        fit_fields(a, args.output_dir)
        summarize_matched(a, args.output_dir)
        for name in ['gene_locus_opening_summary.csv', 'input_provenance.json', 'input_conditions.csv']:
            source = args.baseline_dir / name
            if source.exists(): shutil.copyfile(source, args.output_dir / name)
        log('Saving complete depth-matched atlas')
        a.write_h5ad(target, compression='gzip')
        log('Saved ' + str(target))
    from real_specialization_report import render_report
    render_report(a, grid, args.output_dir)
    log('Depth-matched real-data analysis and visualizations complete')


if __name__ == '__main__':
    main()
