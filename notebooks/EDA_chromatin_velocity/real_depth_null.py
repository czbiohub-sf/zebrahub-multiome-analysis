"""Depth-only diagnostics for real peak specificity (not confidence or p values).

Uniformly sample peaks, hold each peak/celltype's expected CPM constant over time,
and simulate independent Poisson counts at actual versus time-matched exposures.
A third arm binomially thins observed counts to matched exposures. The Poisson
model ignores overdispersion, shared-cell dependence and stage-specific technical
effects; neither arm establishes biological ground truth or corrects the atlas.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts/chrom_velo'))
from real_specialization import parse_condition_grid, compute_block_statistics
from real_input_provenance import PATHS, read_column, plain

METHODS = ('tau', 'entropy')
ARMS = ('constant_cpm_actual_depth', 'constant_cpm_matched_depth', 'observed_thinned_depth')


def metrics(values, floor):
    x = np.asarray(values)[np.isfinite(values)]
    active = np.abs(x) >= floor
    n_active = int(active.sum())
    return dict(n_supported=len(x), median=float(np.median(x)) if len(x) else np.nan,
                mean=float(np.mean(x)) if len(x) else np.nan,
                n_positive=int((x >= floor).sum()), n_negative=int((x <= -floor).sum()),
                n_effect_above_floor=n_active,
                positive_fraction_among_active=float(np.mean(x[active] > 0)) if n_active else np.nan)


def summarize(results, observed, arm, replicate, stages, floor):
    rows = []
    for method in METHODS:
        score, rate = results[method]['score'], results[method]['rate']
        for t, stage in enumerate(stages):
            row = dict(arm=arm, replicate=replicate, method=method, kind='stage_score',
                       support_scope='native', start_hpf=float(stage), end_hpf=float(stage))
            row.update(metrics(score[:, t], floor))
            rows.append(row)
        for t in range(len(stages)-1):
            base = dict(arm=arm, replicate=replicate, method=method, kind='interval_rate',
                        start_hpf=float(stages[t]), end_hpf=float(stages[t+1]))
            rows.append(dict(base, support_scope='native', **metrics(rate[:, t], floor)))
            if arm != 'observed':
                o = observed[method]['rate'][:, t]
                common = np.isfinite(o) & np.isfinite(rate[:, t])
                rows.append(dict(base, support_scope='matched_to_observed',
                                 **metrics(rate[common, t], floor)))
                rows.append(dict(base, support_scope='observed_on_matched_support',
                                 **metrics(o[common], floor)))
    return rows


def compute(counts, coverage, grid):
    return compute_block_statistics(counts * (1e6 / coverage), counts, grid,
                                    min_cells=20, min_total_counts=20,
                                    shrinkage=0.)['fixed_common']


def draw(table, output):
    colors = {'observed':'#ba321d', 'constant_cpm_actual_depth':'#3976b9',
              'constant_cpm_matched_depth':'#23916a', 'observed_thinned_depth':'#a360b4'}
    labels = {'observed':'Observed', 'constant_cpm_actual_depth':'Constant CPM; actual depth',
              'constant_cpm_matched_depth':'Constant CPM; matched depth',
              'observed_thinned_depth':'Observed counts; matched depth'}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for i, method in enumerate(METHODS):
        for j, (kind, statistic, title) in enumerate([
            ('stage_score','median','Median specificity'),
            ('interval_rate','median','Median specificity rate / hpf'),
            ('interval_rate','positive_fraction_among_active','Positive fraction among |rate| ≥ 0.002/hpf')]):
            ax = axes[i,j]
            for arm in ('observed', *ARMS):
                data = table[(table.method == method) & (table.arm == arm) &
                             (table.kind == kind) & (table.support_scope == 'native')]
                for rep, subset in data.groupby('replicate'):
                    x = (subset.start_hpf + subset.end_hpf) / 2
                    ax.plot(x, subset[statistic], 'o-', color=colors[arm],
                            lw=2.3 if arm == 'observed' else 1.1,
                            alpha=1 if arm == 'observed' else .55,
                            label=labels[arm] if rep in (-1, 0) else None)
            ax.set_title(f'{method}: {title}')
            ax.set_xlabel('hpf' if kind == 'stage_score' else 'Interval midpoint, hpf')
            ax.grid(alpha=.2)
            if j == 1:
                ax.axhline(0, color='k', lw=.8, ls=':')
            if j == 2:
                ax.axhline(.5, color='k', lw=.8, ls=':')
                ax.set_ylim(-.02,1.02)
    axes[0,0].legend(fontsize=8, frameon=False)
    fig.suptitle('Depth sensitivity diagnostic; five simulations per arm, not confidence intervals\n'
                 'Constant underlying CPM null ignores overdispersion; native support may differ by arm', fontsize=12)
    fig.savefig(output/'depth_null_diagnostic.png', dpi=180)
    fig.savefig(output/'depth_null_diagnostic.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=PATHS['celltype_state'])
    p.add_argument('--output-dir', type=Path, default=HERE/'outputs/real_specialization_v1')
    p.add_argument('--sample-peaks', type=int, default=8192)
    p.add_argument('--replicates', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--read-block-size', type=int, default=512)
    args = p.parse_args()
    if min(args.sample_peaks, args.replicates, args.read_block_size) < 1:
        p.error('Sampling and block sizes must be positive')
    seeds = np.random.SeedSequence(args.seed).spawn(1+len(ARMS)*args.replicates)
    with h5py.File(args.state, 'r') as f:
        meta = pd.DataFrame({key:read_column(f['var'], key) for key in
                            ('annotation_ML_coarse','dev_stage','n_cells','total_coverage')},
                            index=read_column(f['var'], f['var'].attrs['_index']))
        grid = parse_condition_grid(meta)
        n = f['layers/sum'].shape[0]
        rows = np.sort(np.random.default_rng(seeds[0]).choice(n, min(n,args.sample_peaks), replace=False))
        pieces=[]
        for start in range(0,len(rows),args.read_block_size):
            pieces.append(f['layers/sum'][rows[start:start+args.read_block_size]])
        raw = np.concatenate(pieces).astype(np.float64)
        ids = np.concatenate([f['obs'][f['obs'].attrs['_index']].asstr()[rows[start:start+args.read_block_size]]
                              for start in range(0,len(rows),args.read_block_size)])
    if not np.isfinite(raw).all() or np.any(raw < 0) or not np.array_equal(raw, np.rint(raw)):
        raise ValueError('The observed sum layer must contain finite nonnegative integer counts')
    coverage = meta.total_coverage.to_numpy(float)
    col_ct = np.asarray([list(grid['celltypes']).index(c) for c in meta.annotation_ML_coarse])
    const_rate = np.zeros((len(raw), len(grid['celltypes'])))
    minimum_depth = np.zeros(len(grid['celltypes']))
    for c in range(len(grid['celltypes'])):
        columns = col_ct == c
        const_rate[:,c] = raw[:,columns].sum(axis=1) / coverage[columns].sum()
        minimum_depth[c] = coverage[columns].min()
    matched_coverage = minimum_depth[col_ct]
    expected_actual = const_rate[:,col_ct] * coverage
    expected_matched = const_rate[:,col_ct] * matched_coverage
    assert np.all(matched_coverage <= coverage)
    # Check the defining null invariant before adding measurement noise: the two
    # exposure schedules encode identical CPM and no true temporal specificity.
    np.testing.assert_allclose(expected_actual * (1e6 / coverage),
                               expected_matched * (1e6 / matched_coverage),
                               rtol=1e-12, atol=1e-12)
    noiseless = compute(expected_actual, coverage, grid)
    noiseless_max = {method: float(np.nanmax(np.abs(noiseless[method]['rate'])))
                     for method in METHODS}
    if any(value > 1e-7 for value in noiseless_max.values()):
        raise AssertionError('The constant-CPM null must have zero noiseless rates')
    observed = compute(raw, coverage, grid)
    summary_rows = summarize(observed, observed, 'observed', -1, grid['stages'], .002)
    print(f'Loaded {len(rows)} uniformly sampled peaks; simulating {args.replicates} replicates per arm', flush=True)
    for ai, arm in enumerate(ARMS):
        for rep in range(args.replicates):
            rng = np.random.default_rng(seeds[1+ai*args.replicates+rep])
            if arm == 'constant_cpm_actual_depth':
                counts = rng.poisson(expected_actual); depth = coverage
            elif arm == 'constant_cpm_matched_depth':
                counts = rng.poisson(expected_matched); depth = matched_coverage
            else:
                counts = rng.binomial(raw.astype(np.int64), matched_coverage/coverage); depth = matched_coverage
            result = compute(counts, depth, grid)
            summary_rows.extend(summarize(result, observed, arm, rep, grid['stages'], .002))
        print(f'Completed {arm}', flush=True)
    table = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir/'depth_null_statistics.csv', index=False)
    pd.DataFrame({'global_index':rows, 'peak':ids}).to_csv(args.output_dir/'depth_null_sample_peaks.csv', index=False)
    late = table[(table.kind=='interval_rate') & (table.start_hpf==19)]
    late_summary={}
    for (method,arm,scope), group in late.groupby(['method','arm','support_scope']):
        late_summary[f'{method}/{arm}/{scope}']={key:dict(mean=float(group[key].mean()),
                minimum=float(group[key].min()), maximum=float(group[key].max()))
                for key in ['n_supported','median','mean','positive_fraction_among_active','n_effect_above_floor']}
    fixed = (grid['present'] & (grid['n_cells'] >= 20)).all(axis=1)
    report = dict(source=str(args.state), n_atlas_peaks=n, sampled_peaks=len(rows), sampling='Uniform without replacement, source-row order preserved',
                  seed=args.seed, replicates_per_arm=args.replicates, read_block_size=args.read_block_size,
                  min_cells=20, min_total_counts=20, effect_floor_per_hpf=.002,
                  fixed_celltypes=grid['celltypes'][fixed], stages_hpf=grid['stages'],
                  model='Independent Poisson counts, constant true peak/celltype CPM over stages, fitted by pooled count divided by pooled actual coverage',
                  noiseless_max_abs_specificity_rate=noiseless_max,
                  matched_depth='For each celltype, use minimum of its real condition coverages at all stages',
                  observed_thinning='Independent binomial thinning of each observed count with probability minimum_celltype_coverage / actual_condition_coverage',
                  caveats=['No biological replication or confidence intervals are created.',
                           'The Poisson model ignores overdispersion, cell dependence and stage-specific technical effects.',
                           'Expected constant CPM is fitted from all six observed stages, so this is an explanatory diagnostic, not an independent goodness-of-fit test.',
                           'Each arm applies the same count20 gate; native supported peak subsets can differ. Matched-support rate summaries are also provided.',
                           'Observed thinning is destructive to information only inside an in-memory simulation; source data and primary atlas overlays are unchanged.',
                           'Depth matching cannot separate biological developmental change from stage-specific technical effects.'],
                  late_interval_summary=late_summary)
    (args.output_dir/'depth_null_summary.json').write_text(json.dumps(plain(report),indent=2)+'\n')
    draw(table,args.output_dir)
    print(json.dumps(plain(late_summary),indent=2),flush=True)


if __name__=='__main__':
    main()
