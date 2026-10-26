# %% [markdown]
# # Chromatin velocity: a ground-truth experiment
# Run from the repository root:
# python notebooks/EDA_chromatin_velocity/toy_chromatin_velocity.py
#
# A fixed peak atlas supports signed opening/closing rates directly. Arrows
# between whole time profiles have a different meaning: profile-shift similarity.
# This experiment tests both, including a deliberately simple sign counterexample.

# %% Imports and setup
import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.spatial import cKDTree
import scanpy as sc
import scvelo as scv
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'chrom_velo'))
from temporal_velocity import (STAGES_HPF, temporal_rates, cis_weights,
                               couple_rates, rate_summary, project_scvelo)

PROGRAMS = ['opening', 'closing', 'pulse', 'delayed', 'opposite_lineages', 'constant']


def make_toy(seed=42, n_per_program=90, noise=0.08):
    """Known continuous profiles, independent pseudobulk noise, planted cis modules.

    Graph uses shared simulated regulatory modules, not noisy measured rates.
    This is a favorable denoising benchmark; the rewired graph is its control.
    Two absent columns include an internal gap to test that we never bridge it.
    """
    rng = np.random.default_rng(seed)
    conditions = pd.DataFrame([(c, t) for c in ['neural', 'muscle', 'blood']
                               for t in STAGES_HPF
                               if (c, t) not in [('blood', 10.), ('muscle', 14.)]],
                              columns=['celltype', 'time_hpf'])
    conditions.index = [f'{c}_{t:g}hpf' for c, t in conditions.to_numpy()]
    conditions['n_cells'] = 100
    records, profiles = [], []
    for program in PROGRAMS:
        for k in range(n_per_program):
            # Three chromosomes, each holding several independent modules.
            chromosome = f'chr{k % 3 + 1}'
            target = ['neural', 'muscle', 'blood'][(k // 3) % 3]
            center = rng.normal(16., 0.3)
            amplitude = rng.uniform(0.8, 1.2)
            y = []
            for c, t in conditions[['celltype', 'time_hpf']].to_numpy():
                sigmoid = expit((t - center) / 1.8)
                curve = {'opening': sigmoid, 'closing': 1 - sigmoid,
                         'pulse': np.exp(-0.5 * ((t - center) / 2.5) ** 2),
                         'delayed': expit((t - 21.) / 1.1),
                         'constant': 0.5,
                         'opposite_lineages': sigmoid if c == 'neural' else 1 - sigmoid}[program]
                specificity = 1. if (c == target or program == 'opposite_lineages') else 0.2
                y.append(0.25 + amplitude * specificity * curve)
            profiles.append(y)
            records.append(dict(program=program, chrom=chromosome, target=target,
                                module=f'{program}_{chromosome}_{target}'))
    obs = pd.DataFrame(records)
    obs.index = [f'{c}-{1000*i}-{1000*i+300}' for i, c in enumerate(obs.chrom)]
    truth = np.asarray(profiles)
    x = truth + rng.normal(0., noise, truth.shape)  # no clipping bias
    rows, cols, scores = [], [], []
    for _, group in obs.groupby('module', sort=False):
        idx = obs.index.get_indexer(group.index)
        for i in idx:
            js = rng.choice(idx[idx != i], size=min(5, len(idx)-1), replace=False)
            rows.extend([i] * len(js)); cols.extend(js)
            scores.extend(rng.uniform(0.45, 0.95, len(js)))
    g = sparse.csr_matrix((scores, (rows, cols)), shape=(len(obs), len(obs)))
    g = g.maximum(g.T)
    a = ad.AnnData(x.astype('float32'), obs=obs, var=conditions)
    a.layers['truth_state'] = truth.astype('float32')
    a.obsp['coaccessibility'] = g
    a.uns['simulation'] = dict(seed=seed, noise_sd=noise,
                              missing_conditions=['blood_10hpf', 'muscle_14hpf'])
    return a


def shuffled_cis_graph(g, chrom, rng):
    """Permute graph labels within chromosome: preserves cis restriction/degrees."""
    p = np.arange(len(chrom))
    for c in np.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        p[idx] = rng.permutation(idx)
    return g[p][:, p]


def benchmark(seeds=range(5)):
    rows = []
    for seed in seeds:
        a = make_toy(seed)
        v, mask, _ = temporal_rates(a.X, a.var)
        truth, _, _ = temporal_rates(a.layers['truth_state'], a.var)
        rng = np.random.default_rng(seed + 1000)
        null_graph = shuffled_cis_graph(a.obsp['coaccessibility'], a.obs.chrom.to_numpy(), rng)
        for graph_label, g in [('planted_cis', a.obsp['coaccessibility']), ('shuffled_cis', null_graph)]:
            for threshold in [0.25, 0.5, 0.8]:
                w = cis_weights(g, a.obs.chrom, threshold)
                for alpha in [0., 0.6, 1.]:
                    final, _ = couple_rates(v, mask, w, alpha)
                    dynamic = mask & (np.abs(truth) > 0.015)
                    constant = (a.obs.program == 'constant').to_numpy()[:, None] & mask
                    rows.append(dict(seed=seed, graph=graph_label, threshold=threshold, alpha=alpha,
                                     rmse=float(np.sqrt(np.mean((final[mask]-truth[mask])**2))),
                                     sign_accuracy=float(np.mean(np.sign(final[dynamic]) == np.sign(truth[dynamic]))),
                                     constant_rms=float(np.sqrt(np.mean(final[constant]**2)))))
        # Reorder measured time values within each celltype, preserving missingness.
        shuffled = np.array(a.X, copy=True)
        for c in a.var.celltype.unique():
            idx = np.flatnonzero(a.var.celltype.to_numpy() == c)
            shuffled[:, idx] = shuffled[:, rng.permutation(idx)]
        null_v, _, _ = temporal_rates(shuffled, a.var)
        rows.append(dict(seed=seed, graph='time_permutation', threshold=0.25, alpha=1.,
                         rmse=float(np.sqrt(np.mean((null_v[mask]-truth[mask])**2))),
                         sign_accuracy=float(np.mean(np.sign(null_v[dynamic]) == np.sign(truth[dynamic]))),
                         constant_rms=float(np.sqrt(np.mean(null_v[constant]**2)))))
    return pd.DataFrame(rows)


def add_atlas(a, seed=42):
    # Toy ONLY: compute atlas once from observed state, without using derivatives.
    # For real data these calls must be replaced by the supplied atlas and graph.
    sc.pp.neighbors(a, n_neighbors=20, use_rep='X', random_state=seed)
    sc.tl.umap(a, n_components=2, random_state=seed)
    coords2 = a.obsm['X_umap'].copy()
    sc.tl.umap(a, n_components=3, random_state=seed)
    a.obsm['X_umap3d'] = a.obsm['X_umap'].copy()
    a.obsm['X_umap'] = coords2
    return a


def savefig(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def scatter(ax, coords, color, title, limit=None, cmap='coolwarm'):
    kw = dict(vmin=-limit, vmax=limit) if limit else {}
    obj = ax.scatter(*coords.T, c=color, s=8, cmap=cmap, rasterized=True, **kw)
    ax.set(title=title, xlabel='UMAP 1', ylabel='UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    return obj


def draw_stream(ax, coords, vectors):
    """Interpolate locally and mask grid sites outside occupied neighborhoods."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    gx = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 80)
    gy = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 80)
    xx, yy = np.meshgrid(gx, gy)
    sites = np.c_[xx.ravel(), yy.ravel()]
    tree = cKDTree(coords)
    d, idx = tree.query(sites, k=12)
    radius = np.quantile(tree.query(coords, k=6)[0][:, -1], 0.8)
    weights = np.exp(-0.5 * (d / max(radius, 1e-6)) ** 2)
    velocity = (vectors[idx] * weights[:, :, None]).sum(1) / np.maximum(weights.sum(1)[:, None], 1e-12)
    velocity[d[:, 0] > radius] = np.nan
    grid = velocity.reshape(len(gy), len(gx), 2)
    ax.streamplot(gx, gy, grid[:, :, 0], grid[:, :, 1], color='#25354a',
                  density=1.1, linewidth=0.65, arrowsize=0.7)
    ax.set_xlim(xlim); ax.set_ylim(ylim)


def plot_results(a, mask, metrics, output):
    coords = a.obsm['X_umap']
    summaries = rate_summary(a.layers['velocity'], mask)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    fig.suptitle('Known accessibility programs: positive = opening, negative = closing')
    for ax, program in zip(axes.flat, PROGRAMS):
        i = np.flatnonzero((a.obs.program == program) & (a.obs.target == 'neural'))[0]
        for celltype in a.var.celltype.unique():
            idx = np.flatnonzero(a.var.celltype.to_numpy() == celltype)
            ax.plot(a.var.time_hpf.iloc[idx], a.layers['truth_state'][i, idx], '-o', ms=3, label=celltype)
        ax.set(title=program.replace('_', ' '), xlabel='Time (hpf)', ylabel='Accessibility')
    axes.flat[0].legend(fontsize=8)
    savefig(fig, output, '01_ground_truth_profiles')

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    limit = np.quantile(np.abs(summaries.signed_rate), 0.98)
    p = scatter(axes[0, 0], coords, summaries.signed_rate, 'Mean signed rate (observed intervals)', limit)
    fig.colorbar(p, ax=axes[0, 0], label='Accessibility / hpf')
    p = scatter(axes[0, 1], coords, summaries.magnitude_rms, 'RMS rate: retains transient activity', cmap='viridis')
    fig.colorbar(p, ax=axes[0, 1], label='Accessibility / hpf')
    for ax, title in zip(axes[1], ['Profile-shift arrows (display-normalized)', 'Profile-shift streamlines (local support only)']):
        scatter(ax, coords, summaries.signed_rate, title, limit)
    v = a.obsm['velocity_umap']
    lengths = np.linalg.norm(v, axis=1)
    unit = np.divide(v, lengths[:, None], out=np.zeros_like(v), where=lengths[:, None] > 1e-8)
    keep = np.arange(0, len(a), 3)
    axes[1, 0].quiver(*coords[keep].T, *unit[keep].T, angles='xy', scale_units='xy', scale=2.8,
                      width=0.002, color='#25354a')
    draw_stream(axes[1, 1], coords, v)
    fig.suptitle('One fixed toy peak UMAP: color and arrow direction answer different questions')
    savefig(fig, output, '02_peak_umap_rates_and_arrows')

    # Time-resolved scalar fields: missing transitions are excluded from means.
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), constrained_layout=True)
    limit = np.quantile(np.abs(a.layers['velocity'][mask]), 0.98)
    for ax, start, end in zip(axes, STAGES_HPF[:-1], STAGES_HPF[1:]):
        idx = a.var.time_hpf.to_numpy() == start
        s = rate_summary(a.layers['velocity'][:, idx], mask[:, idx])
        n_ct = int(mask[:, idx].all(axis=0).sum())
        p = scatter(ax, coords, s.signed_rate, f'{start:g} → {end:g} hpf ({n_ct} celltypes)', limit)
    fig.colorbar(p, ax=axes, label='Mean accessibility / hpf', shrink=0.8)
    savefig(fig, output, '03_time_resolved_opening_closing')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    truth = a.layers['truth_velocity']
    axes[0].scatter(truth[mask][::5], a.layers['velocity'][mask][::5], s=3, alpha=0.3)
    axes[0].plot([-0.3, 0.3], [-0.3, 0.3], '--', color='black', lw=1)
    axes[0].set(xlabel='True interval rate', ylabel='Estimated interval rate', title='Ground-truth recovery')
    subset = metrics[(metrics.threshold == 0.25) & (metrics.graph != 'time_permutation')]
    for label, group in subset.groupby('graph'):
        g = group.groupby('alpha').rmse.agg(['mean', 'std'])
        axes[1].errorbar(g.index, g['mean'], yerr=g['std'], marker='o', label=label)
    axes[1].set(xlabel='Intrinsic weight α', ylabel='Rate RMSE (5 seeds)', title='Cis smoothing vs rewired-graph control')
    axes[1].legend(fontsize=8)
    savefig(fig, output, '04_validation')

    xyz, v3 = a.obsm['X_umap3d'], a.obsm['velocity_umap3d']
    norm = np.linalg.norm(v3, axis=1)
    v3 = np.divide(v3, norm[:, None], out=np.zeros_like(v3), where=norm[:, None] > 1e-8)
    fig3 = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
        text=a.obs.program, marker=dict(size=3, color=summaries.signed_rate,
        colorscale='RdBu_r', cmid=0, showscale=True, colorbar=dict(title='Signed rate')),
        name='Peaks'))
    i = np.arange(0, len(a), 4)
    fig3.add_trace(go.Cone(x=xyz[i, 0], y=xyz[i, 1], z=xyz[i, 2], u=v3[i, 0], v=v3[i, 1], w=v3[i, 2],
                           sizemode='absolute', sizeref=0.45, anchor='tail', showscale=False,
                           colorscale=[[0, '#334155'], [1, '#334155']], name='Profile-shift direction'))
    fig3.update_layout(title='Toy 3D peak UMAP · color = opening/closing · cones = profile-shift similarity',
                       scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=55))
    fig3.write_html(output / '05_interactive_3d.html', include_plotlyjs=True)


def phase_counterexample(output):
    """x_tau(t)=f(t-tau): advancing t makes the profile resemble SMALLER tau.

    This analytic family is independent of noisy denoising benchmarks. The x
    axis in the diagnostic below is a known phase coordinate, not a UMAP.
    """
    tau = np.linspace(11., 23., 180)
    t = STAGES_HPF
    x = 0.1 + np.exp(-0.5 * ((t[None, :] - tau[:, None]) / 3.)**2)
    var = pd.DataFrame({'celltype': 'one_lineage', 'time_hpf': t}, index=[f't{j}' for j in range(len(t))])
    a = ad.AnnData(x.astype('float32'), var=var)
    v, mask, _ = temporal_rates(x, var)
    a.layers['velocity'] = v.astype('float32')
    add_atlas(a)
    # Display only; neighborhood selection uses whole measured profiles above.
    a.obsm['X_phase'] = np.c_[tau, np.zeros_like(tau)]
    project_scvelo(a, mask, bases=('phase', 'umap', 'umap3d'))
    vx = a.obsm['velocity_phase'][:, 0]
    interior = (tau > 13) & (tau < 21) & (np.abs(vx) > 1e-8)
    result = {'n_interior_nonzero': int(interior.sum()),
              'fraction_profile_arrows_toward_earlier_program': float(np.mean(vx[interior] < 0)),
              'identity': 'x_tau(t+h) = x_(tau-h)(t)',
              'meaning': 'Positive temporal derivative does not imply early-to-late peak-program arrows'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for phase in [12., 16., 20.]:
        dense_t = np.linspace(8, 27, 200)
        axes[0].plot(dense_t, 0.1 + np.exp(-0.5*((dense_t-phase)/3.)**2), label=f'Peak program at {phase:g} hpf')
    axes[0].set(xlabel='Developmental time (hpf)', ylabel='Accessibility', title='Same pulse, different activation time τ')
    axes[0].legend(fontsize=8)
    keep = np.arange(10, len(tau)-10, 9)
    axes[1].scatter(tau, np.zeros_like(tau), c=tau, s=16, cmap='viridis')
    axes[1].quiver(tau[keep], np.zeros(len(keep)), vx[keep], np.zeros(len(keep)),
                   angles='xy', scale_units='xy', scale=0.6, width=0.006)
    axes[1].set(xlabel='Known program activation time τ (hpf; early → late)', yticks=[], ylim=(-0.5, 0.5),
                title=f'Forward-difference projection: {result["fraction_profile_arrows_toward_earlier_program"]:.0%} point earlier')
    savefig(fig, output, '06_profile_shift_counterexample')
    # A continuous wave of programs on an actual UMAP, using the same graph.
    xy, uv = a.obsm['X_umap'], a.obsm['velocity_umap']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, title in zip(axes, ['Continuous pulse family: profile-shift arrows', 'Locally supported streamlines']):
        color = scatter(ax, xy, tau, title, cmap='viridis')
    lens = np.linalg.norm(uv, axis=1)
    unit = np.divide(uv, lens[:, None], out=np.zeros_like(uv), where=lens[:, None] > 1e-8)
    k = np.arange(0, len(a), 5)
    axes[0].quiver(*xy[k].T, *unit[k].T, angles='xy', scale_units='xy', scale=2.5, width=0.003)
    draw_stream(axes[1], xy, uv)
    fig.colorbar(color, ax=axes, label='Known pulse time τ (hpf)', shrink=0.8)
    fig.suptitle('A continuum of peak programs; these arrows follow profile advance, toward earlier τ')
    savefig(fig, output, '07_continuous_program_umap')
    xyz, uvw = a.obsm['X_umap3d'], a.obsm['velocity_umap3d']
    lengths = np.linalg.norm(uvw, axis=1)
    uvw = np.divide(uvw, lengths[:, None], out=np.zeros_like(uvw), where=lengths[:, None] > 1e-8)
    interactive = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
        marker=dict(size=4, color=tau, colorscale='Viridis', showscale=True, colorbar=dict(title='Pulse hpf')),
        name='Continuous peak programs'))
    interactive.add_trace(go.Cone(x=xyz[k, 0], y=xyz[k, 1], z=xyz[k, 2],
        u=uvw[k, 0], v=uvw[k, 1], w=uvw[k, 2], sizemode='absolute', sizeref=0.35,
        anchor='tail', showscale=False, name='Profile-shift direction'))
    interactive.update_layout(title='Continuous toy peak UMAP: profile-shift direction points toward earlier programs',
                              scene=dict(aspectmode='data'))
    interactive.write_html(output / '08_continuous_program_3d.html', include_plotlyjs=True)
    a.obs['true_pulse_time_hpf'] = tau
    a.layers['velocity_valid'] = mask
    a.write_h5ad(output / 'phase_counterexample.h5ad')
    return result


# %% Execute the reproducible experiment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs' / 'toy_v2')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10})
    a = make_toy()
    v, mask, dt = temporal_rates(a.X, a.var)
    truth, _, _ = temporal_rates(a.layers['truth_state'], a.var)
    w = cis_weights(a.obsp['coaccessibility'], a.obs.chrom)
    final, smoothed = couple_rates(v, mask, w)
    a.layers['velocity_temporal'] = v.astype('float32')
    a.layers['velocity_coupled'] = smoothed.astype('float32')
    a.layers['velocity'] = final.astype('float32')
    a.layers['truth_velocity'] = truth.astype('float32')
    a.layers['velocity_valid'] = mask
    a.var['interval_hpf'] = dt
    summary = rate_summary(final, mask)
    for col in summary:
        a.obs[col] = summary[col].to_numpy()
    print('Computing toy atlas once, then projecting into 2D and 3D.', flush=True)
    add_atlas(a)
    # Leiden labels are a visualization summary; simulated program labels are truth.
    sc.tl.leiden(a, resolution=0.5, key_added='leiden_coarse', random_state=42)
    project_scvelo(a, mask, bases=('umap', 'umap3d'))
    metrics = benchmark()
    metrics.to_csv(output / 'validation_metrics.csv', index=False)
    plot_results(a, mask, metrics, output)
    counterexample = phase_counterexample(output)
    for j in range(2):
        a.obs[f'velocity_umap_{j+1}'] = a.obsm['velocity_umap'][:, j]
    cols = ['signed_rate', 'magnitude_rms', 'opening_rate', 'closing_rate', 'velocity_umap_1', 'velocity_umap_2']
    a.obs.groupby('leiden_coarse', observed=True)[cols].mean().to_csv(output / 'cluster_summary.csv')
    a.obs.to_csv(output / 'peak_summary.csv')
    a.write_h5ad(output / 'toy_chromatin_velocity.h5ad')
    report = {'seed': 42, 'shape': list(a.shape), 'valid_columns': int(mask.all(axis=0).sum()),
              'missing': a.uns['simulation']['missing_conditions'], 'alpha': 0.6, 'threshold': 0.35,
              'phase_counterexample': counterexample,
              'versions': dict(numpy=np.__version__, anndata=ad.__version__, scanpy=sc.__version__, scvelo=scv.__version__)}
    (output / 'run_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print(metrics.groupby(['graph', 'threshold', 'alpha']).rmse.mean().to_string())
    print(f'Outputs: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
